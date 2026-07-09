# ═══════════════════════════════════════════════════════════════
#  train_dnn.py — Train Deep Neural Network (Centralized)
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report, confusion_matrix)
import sys
sys.path.append('src')
from model import DNNModel

print("=" * 55)
print("  DNN TRAINING — Network Traffic Anomaly Detection")
print("=" * 55)

# ── Load cleaned data ──────────────────────────────────────────
print("\n📂 Loading cleaned dataset...")
train_df = pd.read_csv("data/processed/train_cleaned.csv")
test_df  = pd.read_csv("data/processed/test_cleaned.csv")

X_train = torch.tensor(
    train_df.drop(columns=['label']).values, dtype=torch.float32)
y_train = torch.tensor(
    train_df['label'].values, dtype=torch.float32).reshape(-1, 1)
X_test  = torch.tensor(
    test_df.drop(columns=['label']).values,  dtype=torch.float32)
y_test  = torch.tensor(
    test_df['label'].values, dtype=torch.float32).reshape(-1, 1)

print(f"✅ Train : {X_train.shape[0]:,} samples")
print(f"✅ Test  : {X_test.shape[0]:,} samples")
print(f"✅ Features: {X_train.shape[1]}")

# ── DataLoader for batch training ─────────────────────────────
BATCH_SIZE = 512
train_dataset = TensorDataset(X_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True)

# ── Build model ────────────────────────────────────────────────
INPUT_DIM = X_train.shape[1]
model     = DNNModel(INPUT_DIM)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                             weight_decay=1e-4)

# Class weights to handle imbalance (more attacks than normal)
n_normal = (y_train == 0).sum().item()
n_attack = (y_train == 1).sum().item()
pos_weight = torch.tensor([n_normal / n_attack])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Rebuild model without final Sigmoid (BCEWithLogitsLoss needs raw logits)
class DNNModelLogits(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)           # No Sigmoid here — loss handles it
        )
    def forward(self, x):
        return self.network(x)

model     = DNNModelLogits(INPUT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                             weight_decay=1e-4)

# ── Training loop ──────────────────────────────────────────────
EPOCHS = 50
print(f"\n🚀 Training DNN for {EPOCHS} epochs "
      f"(batch size: {BATCH_SIZE})...")
print("-" * 40)

best_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} — Loss: {avg_loss:.4f}")

print("-" * 40)
print("✅ Training complete!")

# ── Evaluate ───────────────────────────────────────────────────
print("\n📊 Evaluating on test set...")
model.eval()
with torch.no_grad():
    logits = model(X_test)
    preds  = (torch.sigmoid(logits) >= 0.5).float().numpy()
    y_true = y_test.numpy()

acc  = accuracy_score(y_true, preds)
prec = precision_score(y_true, preds)
rec  = recall_score(y_true, preds)
f1   = f1_score(y_true, preds)

print("\n" + "=" * 55)
print("  DNN RESULTS")
print("=" * 55)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, preds))
print("\nClassification Report:")
print(classification_report(y_true, preds,
      target_names=['Normal', 'Attack']))

# ── Compare with baselines ─────────────────────────────────────
print("\n" + "=" * 55)
print("  COMPARISON WITH BASELINE MODELS")
print("=" * 55)
print(f"  {'Model':<35} {'Accuracy':>10}")
print(f"  {'-'*45}")
print(f"  {'Logistic Regression (Centralized)':<35} {'93.23%':>10}")
print(f"  {'Random Forest (Centralized)':<35} {'96.07%':>10}")
print(f"  {'Federated LR (FedAvg, 4 clients)':<35} {'84.33%':>10}")
print(f"  {'DNN (Centralized)':<35} {acc*100:>9.2f}%")
print("=" * 55)

# ── Save model ─────────────────────────────────────────────────
torch.save(model.state_dict(), "models/local/dnn_baseline.pt")
print("\n✅ DNN model saved → models/local/dnn_baseline.pt")
print("\n👉 NEXT: Run DNN inside Flower federated learning!")