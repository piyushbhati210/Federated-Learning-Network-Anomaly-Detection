# ═══════════════════════════════════════════════════════════════
#  server.py — Flower Server running FedAvg for DNN or LR
#  Coordinates all 4 clients and combines their models.
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import sys
import argparse
import flwr as fl
import torch
import numpy as np
import pandas as pd

sys.path.append('.')
from model import set_model_parameters

# ── Parse model choice ─────────────────────────────────────────
parser   = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="dnn",
                    choices=["lr", "dnn"],
                    help="Model to use: lr or dnn")
args     = parser.parse_args()
USE_DNN  = args.model == "dnn"

if USE_DNN:
    from model import DNNModel as ModelClass
    MODEL_NAME = "DNN"
    SAVE_PATH  = "../models/global/fedavg_dnn.pt"
else:
    from model import LogisticRegressionModel as ModelClass
    MODEL_NAME = "Logistic Regression"
    SAVE_PATH  = "../models/global/fedavg_logistic_regression.pt"

INPUT_DIM   = 42
NUM_ROUNDS  = 10
MIN_CLIENTS = 4


def get_evaluate_fn():
    """Evaluate global model on shared test set after each round."""
    test_df = pd.read_csv("../data/processed/test_cleaned.csv")
    X_test  = torch.tensor(
        test_df.drop(columns=["label"]).values, dtype=torch.float32)
    y_test  = torch.tensor(
        test_df["label"].values, dtype=torch.float32).reshape(-1, 1)

    def evaluate(server_round, parameters, config):
        model = ModelClass(INPUT_DIM)
        set_model_parameters(model, parameters)
        model.eval()

        with torch.no_grad():
            outputs = model(X_test)

            # Always apply sigmoid to get probabilities (0-1)
            # For LR: sigmoid already applied in forward()
            # For DNN: no sigmoid in forward(), so apply here
            probs = torch.sigmoid(outputs)

            preds    = (probs >= 0.5).float()
            accuracy = (preds == y_test).float().mean().item()

            criterion = torch.nn.BCELoss()
            loss      = criterion(probs, y_test).item()

        print(f"\n🌍 [GLOBAL {MODEL_NAME}] Round {server_round} "
              f"→ Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%\n")

        if server_round == NUM_ROUNDS:
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Final global model saved → {SAVE_PATH}")

        return loss, {"accuracy": accuracy}

    return evaluate


# ── FedAvg strategy ────────────────────────────────────────────
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    evaluate_fn=get_evaluate_fn(),
)

# ── Start server ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"  🌸 FLOWER SERVER — FedAvg with {MODEL_NAME}")
    print(f"  Waiting for {MIN_CLIENTS} clients...")
    print(f"  Total rounds : {NUM_ROUNDS}")
    print("=" * 60)

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )