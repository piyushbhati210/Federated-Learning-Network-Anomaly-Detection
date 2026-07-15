# ═══════════════════════════════════════════════════════════════
#  client_cicids.py — Flower Client for CICIDS2017 + CICIDS2018
#  Supports both LR and DNN models
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

# ── Parse arguments ───────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--client_id", type=int, required=True)
parser.add_argument("--dataset",   type=str, required=True,
                    choices=["cicids2017", "cicids2018"])
parser.add_argument("--model",     type=str, default="dnn",
                    choices=["lr", "dnn"])
args      = parser.parse_args()
CLIENT_ID = args.client_id
DATASET   = args.dataset
USE_DNN   = args.model == "dnn"
MODEL_NAME = "DNN" if USE_DNN else "Logistic Regression"

print(f"\n{'='*55}")
print(f"  Client {CLIENT_ID} — {MODEL_NAME} — {DATASET.upper()}")
print(f"{'='*55}")

# ── Load client data ──────────────────────────────────────────
data_path = f"../data/clients_{DATASET}/client_{CLIENT_ID}_train.csv"
print(f"📂 Loading: {data_path}")
df = pd.read_csv(data_path)

X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values.astype(np.float32).reshape(-1, 1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

print(f"✅ {len(df):,} samples | "
      f"Normal={int((y==0).sum())} | Attack={int((y==1).sum())}")

INPUT_DIM = X.shape[1]
n0 = int((y == 0).sum())
n1 = int((y == 1).sum())

# ── Models ────────────────────────────────────────────────────
class LRModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class DNNModel(nn.Module):
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

# ── Setup model + loss + optimizer ────────────────────────────
if USE_DNN:
    model     = DNNModel(INPUT_DIM)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n0 / max(n1, 1)]))
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=0.001, weight_decay=1e-4)
    EPOCHS = 3
else:
    model     = LRModel(INPUT_DIM)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    EPOCHS = 5

loader = DataLoader(
    TensorDataset(X_tensor, y_tensor),
    batch_size=512, shuffle=True)


def get_params(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_params(model, params):
    sd = dict(zip(model.state_dict().keys(),
                  [torch.tensor(p) for p in params]))
    model.load_state_dict(sd, strict=True)
    return model


def train(model, loader, epochs):
    model.train()
    total = 0
    for _ in range(epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
    return total / (len(loader) * epochs)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        out  = model(X)
        if USE_DNN:
            probs = torch.sigmoid(out)
        else:
            probs = out
        loss = criterion(out if not USE_DNN else out,
                         y).item() if not USE_DNN else \
               nn.BCELoss()(probs, y).item()
        pred = (probs >= 0.5).float()
        acc  = (pred == y).float().mean().item()
    return loss, acc


# ── Flower Client ─────────────────────────────────────────────
class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_params(model)

    def fit(self, parameters, config):
        set_params(model, parameters)
        loss = train(model, loader, EPOCHS)
        print(f"  [Client {CLIENT_ID}] Train loss: {loss:.4f}")
        return get_params(model), len(X_tensor), {}

    def evaluate(self, parameters, config):
        set_params(model, parameters)
        loss, acc = evaluate(model, X_tensor, y_tensor)
        print(f"  [Client {CLIENT_ID}] Eval → "
              f"loss:{loss:.4f} | acc:{acc:.4f}")
        return loss, len(X_tensor), {"accuracy": acc}


if __name__ == "__main__":
    print(f"🌐 Client {CLIENT_ID}: Connecting to server...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FLClient().to_client(),
    )