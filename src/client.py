# ═══════════════════════════════════════════════════════════════
#  client.py — Flower Client for Federated DNN Training
#  Each client trains locally on its OWN data only.
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

sys.path.append('.')
from model import get_model_parameters, set_model_parameters

# ── Parse which client this is (1, 2, 3, or 4) ────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--client_id", type=int, required=True)
parser.add_argument("--model",     type=str, default="dnn",
                    choices=["lr", "dnn"],
                    help="Model to use: lr or dnn")
args      = parser.parse_args()
CLIENT_ID = args.client_id
USE_DNN   = args.model == "dnn"

# ── Import correct model ───────────────────────────────────────
if USE_DNN:
    from model import DNNModel as ModelClass
    MODEL_NAME = "DNN"
else:
    from model import LogisticRegressionModel as ModelClass
    MODEL_NAME = "Logistic Regression"

print(f"\n{'='*50}")
print(f"  Client {CLIENT_ID} — {MODEL_NAME}")
print(f"{'='*50}")

# ── Load this client's own data only ───────────────────────────
print(f"📂 Client {CLIENT_ID}: Loading local data...")
data_path = f"../data/clients/client_{CLIENT_ID}_train.csv"
df        = pd.read_csv(data_path)

X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values.astype(np.float32).reshape(-1, 1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

print(f"✅ Client {CLIENT_ID}: {len(df):,} samples "
      f"(Normal={int((y==0).sum())}, Attack={int((y==1).sum())})")

INPUT_DIM = X.shape[1]

# ── Create model ───────────────────────────────────────────────
model = ModelClass(INPUT_DIM)

# Class weights to handle imbalance
n_normal  = int((y == 0).sum())
n_attack  = int((y == 1).sum())
pos_weight = torch.tensor([n_normal / n_attack])
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.Adam(model.parameters(),
                               lr=0.001, weight_decay=1e-4)

# DataLoader for batch training
dataset = TensorDataset(X_tensor, y_tensor)
loader  = DataLoader(dataset, batch_size=256, shuffle=True)

EPOCHS = 5  # local epochs per FL round


def train(model, loader, epochs):
    model.train()
    total_loss = 0
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / (len(loader) * epochs)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs  = model(X)
        loss     = criterion(outputs, y).item()
        preds    = (torch.sigmoid(outputs) >= 0.5).float()
        accuracy = (preds == y).float().mean().item()
    return loss, accuracy


# ── Flower Client ──────────────────────────────────────────────
class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_model_parameters(model)

    def fit(self, parameters, config):
        set_model_parameters(model, parameters)
        loss = train(model, loader, EPOCHS)
        print(f"  [Client {CLIENT_ID}] Train loss: {loss:.4f}")
        return get_model_parameters(model), len(X_tensor), {}

    def evaluate(self, parameters, config):
        set_model_parameters(model, parameters)
        loss, acc = evaluate(model, X_tensor, y_tensor)
        print(f"  [Client {CLIENT_ID}] Eval → "
              f"loss: {loss:.4f} | accuracy: {acc:.4f}")
        return loss, len(X_tensor), {"accuracy": acc}


# ── Start client ───────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🌐 Client {CLIENT_ID}: Connecting to Flower server...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FLClient().to_client(),
    )