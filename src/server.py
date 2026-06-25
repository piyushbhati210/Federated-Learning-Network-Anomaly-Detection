# ═══════════════════════════════════════════════════════════════
#  server.py — Flower Server running FedAvg
#  Coordinates all 4 clients and combines their models.
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import flwr as fl
import torch
import numpy as np
import pandas as pd

from model import LogisticRegressionModel, set_model_parameters

INPUT_DIM = 42          # number of features in cleaned UNSW-NB15 dataset
NUM_ROUNDS = 10         # how many times clients train + server averages
MIN_CLIENTS = 4         # wait for all 4 clients to connect


def get_evaluate_fn():
    """
    Returns a function the server uses to evaluate the GLOBAL model
    on the shared test set after each round (this is allowed because
    the test set is not private client data — it's our common benchmark).
    """
    test_df = pd.read_csv("../data/processed/test_cleaned.csv")
    X_test = test_df.drop(columns=["label"]).values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32).reshape(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    def evaluate(server_round, parameters, config):
        model = LogisticRegressionModel(INPUT_DIM)
        set_model_parameters(model, parameters)
        model.eval()

        with torch.no_grad():
            outputs = model(X_test_tensor)
            criterion = torch.nn.BCELoss()
            loss = criterion(outputs, y_test_tensor).item()
            predictions = (outputs >= 0.5).float()
            accuracy = (predictions == y_test_tensor).float().mean().item()

        print(f"\n🌍 [GLOBAL MODEL] Round {server_round} "
              f"-> Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}\n")

        # Save the global model after the final round
        if server_round == NUM_ROUNDS:
            torch.save(model.state_dict(), "../models/global/fedavg_logistic_regression.pt")
            print("✅ Final global model saved to models/global/fedavg_logistic_regression.pt")

        return loss, {"accuracy": accuracy}

    return evaluate


# ── Define the FedAvg strategy ──────────────────────────────────
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,                  # use 100% of connected clients each round
    fraction_evaluate=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    evaluate_fn=get_evaluate_fn(),      # evaluate global model on test set each round
)

# ── Start the Flower server ─────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  🌸 STARTING FLOWER SERVER (FedAvg)")
    print(f"  Waiting for {MIN_CLIENTS} clients to connect...")
    print(f"  Total rounds: {NUM_ROUNDS}")
    print("=" * 60)

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )