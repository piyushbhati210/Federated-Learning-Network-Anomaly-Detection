# ═══════════════════════════════════════════════════════════════
#  client.py — Flower Client for Federated Logistic Regression
#  Each client trains locally on its OWN data only.
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import flwr as fl

from model import LogisticRegressionModel, get_model_parameters, set_model_parameters

# ── Parse which client this is (1, 2, 3, or 4) ────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--client_id", type=int, required=True,
                     help="Client number: 1, 2, 3, or 4")
args = parser.parse_args()
CLIENT_ID = args.client_id

DEVICE = torch.device("cpu")

# ── Load this client's own data only ───────────────────────────
print(f"\n📂 Client {CLIENT_ID}: Loading local data...")
data_path = f"../data/clients/client_{CLIENT_ID}_train.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values.astype(np.float32).reshape(-1, 1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

print(f"✅ Client {CLIENT_ID}: Loaded {len(df):,} samples "
      f"(Normal={int((y==0).sum())}, Attack={int((y==1).sum())})")

INPUT_DIM = X.shape[1]

# ── Create local model ──────────────────────────────────────────
model = LogisticRegressionModel(INPUT_DIM).to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train(model, X, y, epochs=1):
    """Train the model locally on this client's data for a few epochs."""
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return loss.item()


def evaluate(model, X, y):
    """Evaluate the model locally on this client's data."""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y).item()
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    return loss, accuracy


# ── Define the Flower Client ────────────────────────────────────
class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_model_parameters(model)

    def fit(self, parameters, config):
        # 1. Receive global weights from server, load into local model
        set_model_parameters(model, parameters)
        # 2. Train locally on THIS client's data only
        loss = train(model, X_tensor, y_tensor, epochs=20)
        print(f"  [Client {CLIENT_ID}] Local training loss: {loss:.4f}")
        # 3. Send updated weights back to server
        return get_model_parameters(model), len(X_tensor), {}

    def evaluate(self, parameters, config):
        set_model_parameters(model, parameters)
        loss, accuracy = evaluate(model, X_tensor, y_tensor)
        print(f"  [Client {CLIENT_ID}] Local eval -> loss: {loss:.4f}, accuracy: {accuracy:.4f}")
        return loss, len(X_tensor), {"accuracy": accuracy}


# ── Start the client and connect to server ──────────────────────
if __name__ == "__main__":
    print(f"\n🌐 Client {CLIENT_ID}: Connecting to Flower server...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FLClient().to_client(),
    )