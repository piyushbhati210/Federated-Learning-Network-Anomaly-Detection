# ═══════════════════════════════════════════════════════════════
#  model.py — Models for Federated Learning (PyTorch)
#  1. LogisticRegressionModel — simple 1-layer model for FedAvg
#  2. DNNModel — Deep Neural Network (3 layers, no sigmoid)
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn


# ── Model 1: Logistic Regression ───────────────────────────────
class LogisticRegressionModel(nn.Module):
    """
    Simple Logistic Regression in PyTorch.
    Used for FedAvg federated learning — has numeric weights
    that can be averaged across clients.
    """
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# ── Model 2: Deep Neural Network ───────────────────────────────
class DNNModel(nn.Module):
    """
    Deep Neural Network for network traffic anomaly detection.
    Architecture: 42 → 128 → 64 → 32 → 1
    - ReLU activations for non-linearity
    - Dropout layers to prevent overfitting
    - NO Sigmoid at output (BCEWithLogitsLoss handles it)
    """
    def __init__(self, input_dim):
        super(DNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), # Layer 1: input → 128 neurons
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),        # Layer 2: 128 → 64 neurons
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),         # Layer 3: 64 → 32 neurons
            nn.ReLU(),
            nn.Linear(32, 1)           # Output: 32 → 1 (raw logit, NO Sigmoid)
        )

    def forward(self, x):
        return self.network(x)


# ── Helper functions for Flower FL ─────────────────────────────
def get_model_parameters(model):
    """Extract model weights as numpy arrays (Flower format)."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model, parameters):
    """Load weights from Flower server into the local model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict  = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return model


# ── Test both models ────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_DIM   = 42
    dummy_input = torch.randn(5, INPUT_DIM)

    # Test Logistic Regression
    print("=" * 45)
    print("Testing Logistic Regression Model")
    print("=" * 45)
    lr_model = LogisticRegressionModel(INPUT_DIM)
    print(lr_model)
    lr_out = lr_model(dummy_input)
    print(f"Output shape : {lr_out.shape}")
    print(f"Sample output: {lr_out.detach().numpy().flatten()}")

    # Test DNN
    print("\n" + "=" * 45)
    print("Testing DNN Model")
    print("=" * 45)
    dnn_model = DNNModel(INPUT_DIM)
    print(dnn_model)
    dnn_out = dnn_model(dummy_input)
    print(f"Output shape : {dnn_out.shape}")
    print(f"Sample output (raw logits): {dnn_out.detach().numpy().flatten()}")
    probs = torch.sigmoid(dnn_out)
    print(f"Sample output (after sigmoid): {probs.detach().numpy().flatten()}")

    print("\n✅ Both models working correctly!")