# ═══════════════════════════════════════════════════════════════
#  server_cicids.py — Flower Server for CICIDS2017 + CICIDS2018
#  Supports LR and DNN models
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import sys
import argparse
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    choices=["cicids2017", "cicids2018"])
parser.add_argument("--model",   type=str, default="dnn",
                    choices=["lr", "dnn"])
args       = parser.parse_args()
DATASET    = args.dataset
USE_DNN    = args.model == "dnn"
MODEL_NAME = "DNN" if USE_DNN else "LR"

INPUT_DIMS = {"cicids2017": 51, "cicids2018": 68}
INPUT_DIM  = INPUT_DIMS[DATASET]
NUM_ROUNDS  = 10
MIN_CLIENTS = 4
SAVE_PATH   = f"../models/global/fedavg_{MODEL_NAME.lower()}_{DATASET}.pt"
TEST_FILE   = f"../data/processed/{DATASET}_test.csv"


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

ModelClass = DNNModel if USE_DNN else LRModel


def set_params(model, params):
    sd = dict(zip(model.state_dict().keys(),
                  [torch.tensor(p) for p in params]))
    model.load_state_dict(sd, strict=True)
    return model


def get_evaluate_fn():
    test_df = pd.read_csv(TEST_FILE)
    X_test  = torch.tensor(
        test_df.drop(columns=["label"]).values,
        dtype=torch.float32)
    y_test  = torch.tensor(
        test_df["label"].values,
        dtype=torch.float32).reshape(-1, 1)

    def evaluate(server_round, parameters, config):
        model = ModelClass(INPUT_DIM)
        set_params(model, parameters)
        model.eval()
        with torch.no_grad():
            out      = model(X_test)
            probs    = torch.sigmoid(out) if USE_DNN else out
            preds    = (probs >= 0.5).float()
            accuracy = (preds == y_test).float().mean().item()
            loss     = nn.BCELoss()(probs, y_test).item()

        print(f"\n🌍 [GLOBAL {MODEL_NAME}-{DATASET.upper()}] "
              f"Round {server_round} → "
              f"Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%\n")

        # ── Log this round to CSV (round-by-round history) ────
        log_path = f"../results/metrics/fedavg_{MODEL_NAME.lower()}_{DATASET}_rounds.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        row = pd.DataFrame([{
            "dataset": DATASET,
            "model": MODEL_NAME,
            "round": server_round,
            "loss": loss,
            "accuracy": accuracy
        }])
        write_header = not os.path.exists(log_path)
        row.to_csv(log_path, mode='a', header=write_header, index=False)

        if server_round == NUM_ROUNDS:
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Final model saved → {SAVE_PATH}")

        return loss, {"accuracy": accuracy}

    return evaluate


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    evaluate_fn=get_evaluate_fn(),
)

if __name__ == "__main__":
    print("=" * 60)
    print(f"  🌸 FLOWER SERVER — FedAvg {MODEL_NAME} on {DATASET.upper()}")
    print(f"  Waiting for {MIN_CLIENTS} clients...")
    print(f"  Rounds: {NUM_ROUNDS} | Input dim: {INPUT_DIM}")
    print("=" * 60)

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )